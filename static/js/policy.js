const policySelect = document.getElementById('policySelect');
const policyStatusBadge = document.getElementById('policyStatusBadge');

document.addEventListener('DOMContentLoaded', function() {
    console.log("Policy selector initializing...");
    loadPolicyOptions();
    policySelect.addEventListener('change', switchPolicy);
    console.log("Policy selector initialized");
});

function loadPolicyOptions() {
    fetch('/api/policy')
        .then(response => response.json())
        .then(data => {
            const activePolicy = data.active;
            const availablePolicies = data.available;

            policySelect.innerHTML = availablePolicies
                .map(policy => `<option value="${policy}">${policy}</option>`)
                .join('');

            policySelect.value = activePolicy;
            updatePolicyBadge(data.model_loaded, data.fallback_active, activePolicy);
        })
        .catch(error => {
            console.error('Error loading policy options:', error);
            policySelect.innerHTML = '<option value="">Error loading policies</option>';
        });
}

function switchPolicy() {
    const selectedPolicy = policySelect.value;
    if (!selectedPolicy) {
        return;
    }

    fetch('/api/policy', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ policy: selectedPolicy })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`Switched to policy: ${selectedPolicy}`);
                updatePolicyBadge(data.model_loaded, data.fallback_active, selectedPolicy);
            } else {
                console.error('Failed to switch policy:', data.error);
                loadPolicyOptions();
            }
        })
        .catch(error => {
            console.error('Error switching policy:', error);
            loadPolicyOptions();
        });
}

function updatePolicyBadge(modelLoaded, fallbackActive, policy) {
    if (policy !== 'rl') {
        policyStatusBadge.textContent = '';
        return;
    }

    if (modelLoaded && !fallbackActive) {
        policyStatusBadge.textContent = '✓ Model loaded';
        policyStatusBadge.style.color = '#4CAF50';
    } else if (fallbackActive) {
        policyStatusBadge.textContent = '⚠ Using greedy fallback';
        policyStatusBadge.style.color = '#FF9800';
    } else {
        policyStatusBadge.textContent = '✗ Model not available';
        policyStatusBadge.style.color = '#F44336';
    }
}
