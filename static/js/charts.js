let waitTimeChart;
let queueLengthChart;

const maxDataPoints = 30;
const waitTimeData = new Array(maxDataPoints).fill(0);
const queueLengthData = new Array(maxDataPoints).fill(0);
const labels = new Array(maxDataPoints).fill('');

document.addEventListener('DOMContentLoaded', function() {
    initCharts();
});

// Create the charts
function initCharts() {
    waitTimeData.fill(0);
    queueLengthData.fill(0);
    
    // Wait Time Chart
    const waitTimeCtx = document.getElementById('waitTimeChart').getContext('2d');
    if (waitTimeChart) waitTimeChart.destroy();
    
    waitTimeChart = new Chart(waitTimeCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Wait Time (min)',
                data: waitTimeData,
                fill: false,
                borderColor: '#F44336',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Minutes'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
    
    // Queue Length Chart
    const queueLengthCtx = document.getElementById('queueLengthChart').getContext('2d');
    if (queueLengthChart) queueLengthChart.destroy();
    
    queueLengthChart = new Chart(queueLengthCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Max Queue Length',
                data: queueLengthData,
                fill: false,
                borderColor: '#2196F3',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'EVs in Queue'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

function updateCharts(data) {
    // Shift data out if we've reached max points
    if (waitTimeData.length >= maxDataPoints) {
        waitTimeData.shift();
        queueLengthData.shift();
        labels.shift();
    }
    
    // Add new data points
    const waitTimeMinutes = data.metrics.average_wait_time / 60;
    waitTimeData.push(waitTimeMinutes);
    
    queueLengthData.push(data.metrics.max_queue_length);
    
    // Add step number as label
    labels.push(data.step);
    
    // Update charts
    waitTimeChart.update();
    queueLengthChart.update();
}