import os

from flask import Flask, render_template, jsonify, request, send_from_directory
import config
import argparse
from utils.data_generator import generate_synthetic_data
from models.simulation import Simulation
from ml.policies.registry import get_policy, list_policies
from scripts.render_markdown import render_markdown

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


class _TrustedHostsMiddleware:
    _allowed = {
        'evcharge.duckdns.org',
        'localhost',
        '127.0.0.1',
        '::1',
        '3.108.5.112',
    }

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        host = environ.get('HTTP_HOST', '').split(':')[0].lower()
        if host not in self._allowed:
            start_response('400 Bad Request', [('Content-Type', 'text/plain')])
            return [b'Bad Request']
        return self.app(environ, start_response)


parser = argparse.ArgumentParser(description='EV Queue Simulation Server')
parser.add_argument('--no-cache', action='store_true', help='Disable data caching')
parser.add_argument('--clear-cache', action='store_true', help='Clear existing cache before starting')
args, _ = parser.parse_known_args()

app = Flask(__name__)
app.wsgi_app = _TrustedHostsMiddleware(app.wsgi_app)
config.validate_required_config()

if args.clear_cache:
    import os
    import glob
    print("Clearing cache files...")
    for cache_file in glob.glob("*.pkl"):
        try:
            os.remove(cache_file)
            print(f"Removed {cache_file}")
        except Exception as e:
            print(f"Failed to remove {cache_file}: {e}")

def initialize_simulation():
    """Initialize simulation data unless explicitly disabled."""
    if not config.BOOTSTRAP_SIMULATION:
        print("Skipping simulation bootstrap (BOOTSTRAP_SIMULATION=false)")
        return [], [], [], Simulation([], [], [])

    print("Initializing simulation data...")
    evs_local, stations_local, routes_local = generate_synthetic_data(
        100, 20, 80, 240, use_cache=not args.no_cache
    )
    print("Creating simulation engine...")
    policy = get_policy(config.ASSIGNMENT_POLICY)
    simulation_local = Simulation(evs_local, stations_local, routes_local, policy)
    print("Server initialization complete!")
    return evs_local, stations_local, routes_local, simulation_local


evs, stations, routes, simulation = initialize_simulation()


def load_policy_comparison():
    """Load high-density policy comparison from results/_agg.json."""
    comparison_file = os.path.join(RESULTS_DIR, '_agg.json')
    if not os.path.isfile(comparison_file):
        return []

    try:
        import json
        with open(comparison_file, 'r') as f:
            data = json.load(f)

        high_density = [row for row in data if row.get('scenario') == 'high']
        selected_policies = {row['policy']: row for row in high_density if row['policy'] in ['greedy', 'nearest', 'rl']}

        return [
            {
                'policy': policy,
                'average_wait_time': f"{int(row['average_wait_time'] / 60)}:{int(row['average_wait_time'] % 60):02d}",
                'avg_queue_length': f"{row['avg_queue_length']:.1f}",
                'completion_rate': f"{row['completion_rate'] * 100:.1f}%",
                'avg_station_utilization': f"{row['avg_station_utilization']:.1%}",
            }
            for policy, row in [(p, selected_policies[p]) for p in ['greedy', 'nearest', 'rl'] if p in selected_policies]
        ]
    except Exception:
        return []


@app.route('/')
def index():
    policy_comparison = load_policy_comparison()
    return render_template('index.html', api_key=config.GOOGLE_MAPS_API_KEY, policy_comparison=policy_comparison)

@app.route('/health')
def health():
    """Health check endpoint for load balancers"""
    return {"status": "healthy"}, 200

@app.route('/results')
def results_page():
    def read_and_render(filename):
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return render_markdown(f.read())

    comparison_html = read_and_render('comparison_summary.md')
    stats_html = read_and_render('stats_summary.md')
    return render_template(
        'results.html',
        comparison_html=comparison_html,
        stats_html=stats_html,
    )

@app.route('/results/assets/<path:filename>')
def results_assets(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    success = simulation.start()
    return jsonify({'success': success})

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    success = simulation.stop()
    return jsonify({'success': success})

@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    try:
        success = simulation.reset()
        return jsonify({'success': success})
    except Exception as exc:
        app.logger.exception("Failed to reset simulation")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/simulation/state')
def get_state():
    state = simulation.get_current_state()
    if not state:
        return jsonify({'status': 'idle'})
    return jsonify(state)

@app.route('/api/simulation/history')
def get_history():
    start = int(request.args.get('start', 0))
    count = int(request.args.get('count', 100))
    history = simulation.get_history(start, count)
    return jsonify(history)

@app.route('/api/optimization/logs')
def get_optimization_logs():
    logs = simulation.get_optimization_logs()
    return jsonify({'logs': logs})

@app.route('/api/ev/journey-log/<ev_id>')
def get_ev_journey_log(ev_id):
    """Get detailed journey log for a specific EV"""
    journey_log = simulation.get_ev_journey_log(ev_id)
    return jsonify({'ev_id': ev_id, 'journey_log': journey_log})


@app.route('/api/policy')
def get_policy_info():
    policy = simulation.policy
    model_loaded = getattr(policy, 'model_loaded', False)
    fallback_active = getattr(policy, 'fallback_active', False)
    return jsonify({
        'active': simulation.policy_name,
        'available': list_policies(),
        'model_loaded': model_loaded,
        'fallback_active': fallback_active,
    })


@app.route('/api/policy', methods=['POST'])
def set_policy():
    """Hot-swap assignment policy."""
    data = request.get_json(silent=True) or {}
    policy_name = data.get('policy')
    if not policy_name:
        return jsonify({'error': 'policy field required'}), 400

    available = list_policies()
    if policy_name not in available:
        return jsonify({'error': f'unknown policy: {policy_name}'}), 400

    new_policy = get_policy(policy_name)
    simulation.set_policy(new_policy)
    return jsonify({
        'success': True,
        'active': simulation.policy_name,
        'model_loaded': getattr(new_policy, 'model_loaded', False),
        'fallback_active': getattr(new_policy, 'fallback_active', False),
    })

@app.route('/api/stations')
def get_stations():
    return jsonify([station.to_dict() for station in stations])

@app.route('/api/evs')
def get_evs():
    return jsonify([ev.to_dict() for ev in evs])

@app.route('/api/routes')
def get_routes():
    return jsonify(routes)

@app.route('/api/generate', methods=['POST'])
def regenerate_data():
    global evs, stations, routes, simulation
    
    simulation.stop()
    
    num_evs = int(request.json.get('num_evs', 100))
    num_stations = int(request.json.get('num_stations', 20))
    num_nodes = int(request.json.get('num_nodes', 80))
    num_routes = int(request.json.get('num_routes', 240))
    use_cache = request.json.get('use_cache', True)
    
    print(f"Regenerating data with {num_evs} EVs, {num_stations} stations, {num_nodes} nodes, {num_routes} routes...")
    print(f"Cache usage: {'enabled' if use_cache else 'disabled'}")
    
    evs, stations, routes = generate_synthetic_data(num_evs, num_stations, num_nodes, num_routes, use_cache=use_cache)
    
    print("Creating new simulation engine...")
    policy = get_policy(config.ASSIGNMENT_POLICY)
    simulation = Simulation(evs, stations, routes, policy)
    print("Regeneration complete!")
    
    return jsonify({
        'success': True, 
        'num_evs': num_evs, 
        'num_stations': num_stations,
        'num_nodes': num_nodes,
        'num_routes': num_routes,
        'cache_used': use_cache
    })

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
