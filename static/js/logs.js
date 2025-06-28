let lastLogCount = 0;

const optimizationLogsEl = document.getElementById('optimizationLogs');

document.addEventListener('DOMContentLoaded', function() {
    // Initial update of logs
    updateOptimizationLogs();
});

function updateOptimizationLogs() {
    if (simulationRunning) {
        fetch('/api/optimization/logs')
        .then(response => response.json())
        .then(data => {
            if (data.logs && data.logs.length > 0) {
                // Display logs
                optimizationLogsEl.textContent = data.logs.join('\n');
                
                // Auto-scroll to bottom if logs have been updated
                if (data.logs.length !== lastLogCount) {
                    lastLogCount = data.logs.length;
                    optimizationLogsEl.scrollTop = optimizationLogsEl.scrollHeight;
                }
            } else {
                optimizationLogsEl.textContent = "No optimization logs yet.";
            }
        })
        .catch(error => console.error('Error updating optimization logs:', error));
    }
}

// Update the main.js updateSimulationState function to also update logs
const originalUpdateSimulationState = updateSimulationState;
updateSimulationState = function() {
    originalUpdateSimulationState();
    
    // Update logs every few updates to avoid too many requests
    stateUpdateCounter++;
    if (stateUpdateCounter % 10 === 0) {
        updateOptimizationLogs();
    }
};