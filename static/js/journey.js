let evSelector;
let journeyTimeline;

let currentSelectedEV = null;
let evList = [];

document.addEventListener('DOMContentLoaded', function() {
    console.log("Journey module initializing...");
    
    // Get DOM elements
    evSelector = document.getElementById('evSelector');
    journeyTimeline = document.getElementById('journeyTimeline');
    
    if (!evSelector) {
        console.error("EV selector dropdown not found in the DOM!");
        return;
    }
    
    console.log("EV selector found:", evSelector);
    
    // Setup EV selector
    evSelector.addEventListener('change', function() {
        currentSelectedEV = this.value;
        console.log("Selected EV:", currentSelectedEV);
        updateJourneyTimeline();
    });
    
    // Initial load of EVs for selector
    updateEVSelector();
    
    console.log("Journey module initialized");
});

function updateEVSelector() {
    console.log("Updating EV selector...");
    
    if (!evSelector) {
        evSelector = document.getElementById('evSelector');
        if (!evSelector) {
            console.error("EV selector still not found!");
            return;
        }
    }
    
    fetch('/api/evs')
    .then(response => response.json())
    .then(evs => {
        console.log(`Got ${evs.length} EVs from server`);
        evList = evs;
        
        // Clear existing options
        evSelector.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select an EV';
        evSelector.appendChild(defaultOption);
        
        // Add options for each EV
        evs.forEach(ev => {
            const option = document.createElement('option');
            option.value = ev.id;
            
            // Add icon/status to option text
            let status = 'Driving';
            if (ev.trip_completed) {
                status = 'Completed';
            } else if (ev.abandoned) {
                status = 'Abandoned';
            } else if (ev.charging) {
                status = 'Charging';
            } else if (ev.in_queue) {
                status = 'In Queue';
            }
            
            option.textContent = `${ev.id} - ${status} - Battery: ${(ev.soc * 100).toFixed(1)}%`;
            evSelector.appendChild(option);
        });
        
        console.log(`Added ${evs.length} options to selector`);
        
        // If a previous selection exists, try to restore it
        if (currentSelectedEV) {
            evSelector.value = currentSelectedEV;
            // If the selected EV no longer exists, reset
            if (evSelector.value !== currentSelectedEV) {
                currentSelectedEV = null;
                journeyTimeline.innerHTML = '<p class="no-ev-selected">Select an EV to view its journey timeline.</p>';
            }
        }
    })
    .catch(error => console.error('Error loading EVs for selector:', error));
}

function updateJourneyTimeline() {
    if (!journeyTimeline) {
        journeyTimeline = document.getElementById('journeyTimeline');
        if (!journeyTimeline) {
            console.error("Journey timeline element not found!");
            return;
        }
    }
    
    if (!currentSelectedEV) {
        journeyTimeline.innerHTML = '<p class="no-ev-selected">Select an EV to view its journey timeline.</p>';
        return;
    }
    
    console.log("Fetching journey log for EV:", currentSelectedEV);
    
    fetch(`/api/ev/journey-log/${currentSelectedEV}`)
    .then(response => response.json())
    .then(data => {
        if (!data.journey_log || data.journey_log.length === 0) {
            journeyTimeline.innerHTML = '<p class="no-ev-selected">No journey logs available for this EV.</p>';
            return;
        }
        
        console.log(`Got ${data.journey_log.length} journey log entries`);
        
        // Clear existing timeline
        journeyTimeline.innerHTML = '';
        
        // Add events to timeline
        data.journey_log.forEach(event => {
            const eventElement = createEventElement(event);
            journeyTimeline.appendChild(eventElement);
        });
        
        // Scroll to bottom to show most recent events
        journeyTimeline.scrollTop = journeyTimeline.scrollHeight;
    })
    .catch(error => console.error('Error loading journey log:', error));
}

function createEventElement(event) {
    const eventElement = document.createElement('div');
    eventElement.className = 'timeline-event';
    
    // Add event-specific class based on event type
    switch (event.event) {
        case 'Initialized':
            eventElement.classList.add('event-info');
            break;
        case 'Moved':
            eventElement.classList.add('event-moved');
            break;
        case 'Started Charging':
        case 'Charging Progress':
        case 'Charging Complete':
            eventElement.classList.add('event-charging');
            break;
        case 'Joined Queue':
            eventElement.classList.add('event-info');
            break;
        case 'Insufficient Battery':
        case 'Charging Needed':
            eventElement.classList.add('event-warning');
            break;
        case 'Abandoned':
            eventElement.classList.add('event-abandoned');
            break;
        case 'Trip Completed':
            eventElement.classList.add('event-info');
            break;
        default:
            eventElement.classList.add('event-info');
    }
    
    // Format timestamp
    const timestamp = new Date(event.timestamp);
    const formattedTime = timestamp.toLocaleTimeString();
    
    // Create event content
    const timeElement = document.createElement('div');
    timeElement.className = 'event-time';
    timeElement.textContent = formattedTime;
    
    const titleElement = document.createElement('div');
    titleElement.className = 'event-title';
    titleElement.textContent = event.event;
    
    const detailsElement = document.createElement('div');
    detailsElement.className = 'event-details';
    
    // Add each detail as a separate line
    Object.entries(event.details).forEach(([key, value]) => {
        const detailLine = document.createElement('div');
        detailLine.className = 'event-detail';
        
        const label = document.createElement('span');
        label.className = 'event-detail-label';
        label.textContent = formatLabel(key) + ':';
        
        const valueSpan = document.createElement('span');
        valueSpan.textContent = value;
        
        detailLine.appendChild(label);
        detailLine.appendChild(valueSpan);
        detailsElement.appendChild(detailLine);
    });
    
    // Assemble event element
    eventElement.appendChild(timeElement);
    eventElement.appendChild(titleElement);
    eventElement.appendChild(detailsElement);
    
    return eventElement;
}

function formatLabel(key) {
    // Convert camelCase or snake_case to Title Case With Spaces
    return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();
}

// Make sure updateEVSelector is available globally
window.updateEVSelector = updateEVSelector;
window.updateJourneyTimeline = updateJourneyTimeline;

// Add a temporary call to try to initialize after a delay (fallback)
setTimeout(() => {
    console.log("Delayed initialization check...");
    if (!evSelector) {
        evSelector = document.getElementById('evSelector');
        if (evSelector) {
            console.log("Found EV selector on delayed check, setting up...");
            evSelector.addEventListener('change', function() {
                currentSelectedEV = this.value;
                updateJourneyTimeline();
            });
            updateEVSelector();
        }
    }
}, 1000);