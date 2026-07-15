document.addEventListener("DOMContentLoaded", () => {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const videoStream = document.getElementById('video-stream');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    const maskCount = document.getElementById('mask-count');
    const noMaskCount = document.getElementById('no-mask-count');
    const confidenceVal = document.getElementById('confidence-val');

    let statsInterval = null;

    function fetchStats() {
        fetch('/stats')
            .then(res => res.json())
            .then(data => {
                maskCount.textContent = data.masks;
                noMaskCount.textContent = data.no_masks;
                confidenceVal.textContent = data.confidence + "%";
            })
            .catch(err => console.error(err));
    }

    btnStart.addEventListener('click', () => {
        // Send request to backend to initialize camera
        fetch('/start_stream', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    // Update UI
                    videoPlaceholder.style.display = 'none';
                    videoStream.style.display = 'block';
                    // Append timestamp to force browser to reload stream (avoid cache)
                    videoStream.src = `/video_feed?t=${new Date().getTime()}`;
                    
                    btnStart.disabled = true;
                    btnStop.disabled = false;
                    
                    statusDot.classList.remove('offline');
                    statusDot.classList.add('online');
                    statusText.textContent = 'Camera Online - Detecting...';

                    // Start fetching stats every 500ms
                    if (statsInterval) clearInterval(statsInterval);
                    statsInterval = setInterval(fetchStats, 500);
                }
            });
    });

    btnStop.addEventListener('click', () => {
        // Send request to backend to stop camera
        fetch('/stop_stream', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'stopped') {
                    // Update UI
                    videoStream.src = "";
                    videoStream.style.display = 'none';
                    videoPlaceholder.style.display = 'flex';
                    
                    btnStart.disabled = false;
                    btnStop.disabled = true;
                    
                    statusDot.classList.remove('online');
                    statusDot.classList.add('offline');
                    statusText.textContent = 'Camera Offline';

                    // Stop fetching stats
                    if (statsInterval) clearInterval(statsInterval);
                    maskCount.textContent = "0";
                    noMaskCount.textContent = "0";
                    confidenceVal.textContent = "0.00%";
                }
            });
    });
});
