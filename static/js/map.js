let map;
let evMarkers = {};
let stationMarkers = {};

function initMap() {
    // Center on Bangalore
    const bangalore = { lat: 12.9716, lng: 77.5946 };
    
    // Create map
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 13,
        center: bangalore,
        mapTypeId: google.maps.MapTypeId.ROADMAP,
        styles: [
            {
                featureType: 'poi',
                stylers: [{ visibility: 'off' }]
            }
        ]
    });
    
    // Initial load of stations and EVs
    loadMapData();
}

function loadMapData() {
    fetch('/api/stations')
    .then(response => response.json())
    .then(stations => {
        stations.forEach(station => {
            createStationMarker(station);
        });
    })
    .catch(error => console.error('Error loading stations:', error));
    
    fetch('/api/evs')
    .then(response => response.json())
    .then(evs => {
        evs.forEach(ev => {
            createEVMarker(ev);
        });
    })
    .catch(error => console.error('Error loading EVs:', error));
}

function createStationMarker(station) {
    const position = {
        lat: station.location[0],
        lng: station.location[1]
    };
    
    const marker = new google.maps.Marker({
        position: position,
        map: map,
        title: `Station ${station.id} (${station.num_chargers} chargers)`,
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: '#4CAF50',
            fillOpacity: 0.8,
            strokeColor: '#388E3C',
            strokeWeight: 2
        },
        zIndex: 10
    });
    
    const infoContent = `
        <div>
            <h3>Station ${station.id}</h3>
            <p>Chargers: ${station.num_chargers}</p>
            <p>Charging Rate: ${station.charging_rate} kW</p>
            <p>Queue: ${station.queue_length}</p>
        </div>
    `;
    
    const infoWindow = new google.maps.InfoWindow({
        content: infoContent
    });
    
    marker.addListener('click', () => {
        infoWindow.open(map, marker);
    });
    
    stationMarkers[station.id] = {
        marker: marker,
        infoWindow: infoWindow
    };
}

function createEVMarker(ev) {
    const position = {
        lat: ev.current_position[0],
        lng: ev.current_position[1]
    };
    
    // Color based on EV state
    let fillColor = '#2196F3'; // Default blue
    if (ev.charging) {
        fillColor = '#4CAF50'; // Green when charging
    } else if (ev.in_queue) {
        fillColor = '#FFC107'; // Yellow when in queue
    } else if (ev.soc < 0.2) {
        fillColor = '#F44336'; // Red when low battery
    }
    
    const marker = new google.maps.Marker({
        position: position,
        map: map,
        title: `EV ${ev.id} (${(ev.soc * 100).toFixed(1)}%)`,
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 7,
            fillColor: fillColor,
            fillOpacity: 0.8,
            strokeColor: '#000000',
            strokeWeight: 1
        },
        zIndex: 5
    });
    
    const infoContent = `
        <div>
            <h3>EV ${ev.id}</h3>
            <p>Battery: ${(ev.soc * 100).toFixed(1)}%</p>
            <p>Status: ${ev.charging ? 'Charging' : (ev.in_queue ? 'In Queue' : 'Driving')}</p>
            ${ev.assigned_station ? `<p>Assigned to: Station ${ev.assigned_station}</p>` : ''}
            ${ev.waiting_time > 0 ? `<p>Wait time: ${ev.waiting_time}s</p>` : ''}
        </div>
    `;
    
    const infoWindow = new google.maps.InfoWindow({
        content: infoContent
    });
    
    marker.addListener('click', () => {
        infoWindow.open(map, marker);
    });
    
    evMarkers[ev.id] = {
        marker: marker,
        infoWindow: infoWindow
    };
}

function updateMapMarkers(evs, stations) {
    evs.forEach(ev => {
        const markerInfo = evMarkers[ev.id];
        
        if (markerInfo) {
            // Update position
            markerInfo.marker.setPosition({
                lat: ev.current_position[0],
                lng: ev.current_position[1]
            });
            
            // Update color based on EV state
            let fillColor = '#2196F3'; // Default blue
            if (ev.charging) {
                fillColor = '#4CAF50'; // Green when charging
            } else if (ev.in_queue) {
                fillColor = '#FFC107'; // Yellow when in queue
            } else if (ev.soc < 0.2) {
                fillColor = '#F44336'; // Red when low battery
            }
            
            markerInfo.marker.setIcon({
                path: google.maps.SymbolPath.CIRCLE,
                scale: 7,
                fillColor: fillColor,
                fillOpacity: 0.8,
                strokeColor: '#000000',
                strokeWeight: 1
            });
            
            // Update info window content
            const infoContent = `
                <div>
                    <h3>EV ${ev.id}</h3>
                    <p>Battery: ${(ev.soc * 100).toFixed(1)}%</p>
                    <p>Status: ${ev.charging ? 'Charging' : (ev.in_queue ? 'In Queue' : 'Driving')}</p>
                    ${ev.assigned_station ? `<p>Assigned to: Station ${ev.assigned_station}</p>` : ''}
                    ${ev.waiting_time > 0 ? `<p>Wait time: ${ev.waiting_time}s</p>` : ''}
                </div>
            `;
            
            markerInfo.infoWindow.setContent(infoContent);
        } else {
            // Create new marker if not exists
            createEVMarker(ev);
        }
    });
    
    stations.forEach(station => {
        const markerInfo = stationMarkers[station.id];
        
        if (markerInfo) {
            // Update info window content
            const infoContent = `
                <div>
                    <h3>Station ${station.id}</h3>
                    <p>Chargers: ${station.num_chargers}</p>
                    <p>Charging Rate: ${station.charging_rate} kW</p>
                    <p>Queue: ${station.queue_length}</p>
                    <p>EVs Charging: ${station.charging_evs.length}</p>
                </div>
            `;
            
            markerInfo.infoWindow.setContent(infoContent);
        } else {
            // Create new marker if not exists
            createStationMarker(station);
        }
    });
}

function updateMap() {
    evMarkers = {};
    stationMarkers = {};

    loadMapData();
}