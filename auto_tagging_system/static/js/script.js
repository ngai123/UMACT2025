// static/js/script.js
// Enhanced: Typing animation, animated bars, interactive UI

document.addEventListener('DOMContentLoaded', () => {
    const complaintTextArea = document.getElementById('complaintText');
    const predictButton = document.getElementById('predictButton');
    const resultsSection = document.getElementById('resultsSection');
    const submittedComplaintTextElem = document.getElementById('submittedComplaintText');
    const predictedProductElem = document.getElementById('predictedProduct');
    const confidenceScoreElem = document.getElementById('confidenceScore');
    const confidenceBarInner = document.getElementById('confidenceBarInner');
    const allProbabilitiesContainer = document.getElementById('allProbabilitiesContainer');
    const allProbabilitiesListElem = document.getElementById('allProbabilitiesList');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorSection = document.getElementById('errorSection');
    const errorMessageElem = document.getElementById('errorMessage');
    const sentimentCard = document.getElementById('sentimentCard');
    const sentimentEmoji = document.getElementById('sentimentEmoji');
    const sentimentLabel = document.getElementById('sentimentLabel');
    const sentimentScoreBar = document.getElementById('sentimentScoreBar');
    const sentimentScoreValue = document.getElementById('sentimentScoreValue');

    // Typing animation for input
    complaintTextArea.addEventListener('input', () => {
        if (!document.querySelector('.typing-cursor')) {
            complaintTextArea.parentElement.insertAdjacentHTML('beforeend', '<span class="typing-cursor">|</span>');
        }
        clearTimeout(complaintTextArea._typingTimeout);
        document.querySelector('.typing-cursor').style.display = 'inline-block';
        complaintTextArea._typingTimeout = setTimeout(() => {
            if (document.querySelector('.typing-cursor')) {
                document.querySelector('.typing-cursor').style.display = 'none';
            }
        }, 800);
    });

    predictButton.addEventListener('click', async () => {
        const complaintText = complaintTextArea.value.trim();

        if (!complaintText) {
            displayError("Please enter some complaint text.");
            return;
        }

        // Show loading indicator and hide previous results/errors
        if (loadingIndicator) loadingIndicator.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';
        if (errorSection) errorSection.style.display = 'none';
        if (allProbabilitiesContainer) allProbabilitiesContainer.style.display = 'none';
        if (allProbabilitiesListElem) allProbabilitiesListElem.innerHTML = '';
        if (sentimentCard) sentimentCard.style.display = 'none';
        const mascotQuote = document.getElementById('mascotQuote');
        if (mascotQuote) mascotQuote.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ complaint_text: complaintText }),
            });

            if (loadingIndicator) loadingIndicator.style.display = 'none';

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response from server.' }));
                throw new Error(errorData.error || `Server responded with status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                displayError(data.error);
            } else if (data.predicted_product) {
                if (submittedComplaintTextElem) await typeText(submittedComplaintTextElem, complaintText, 18);
                if (predictedProductElem) await typeText(predictedProductElem, data.predicted_product, 22);
                animateConfidenceBar(data.confidence);
                if (confidenceScoreElem) confidenceScoreElem.textContent = `${(data.confidence * 100).toFixed(2)}%`;
                if (resultsSection) resultsSection.style.display = 'block';
                // Show mascot/quote
                if (mascotQuote) mascotQuote.style.display = 'flex';
                // Sentiment display
                if (data.sentiment) {
                    if (sentimentCard) sentimentCard.style.display = 'flex';
                    if (sentimentCard) sentimentCard.classList.remove('fadeInBounce');
                    if (sentimentCard) void sentimentCard.offsetWidth;
                    if (sentimentCard) sentimentCard.classList.add('fadeInBounce');
                    if (sentimentEmoji) sentimentEmoji.textContent = data.sentiment.emoji;
                    if (sentimentLabel) sentimentLabel.textContent = `Sentiment: ${capitalizeFirst(data.sentiment.sentiment)}`;
                    const percent = ((data.sentiment.mean_score + 1) / 2) * 100;
                    if (sentimentScoreBar) sentimentScoreBar.style.width = percent + '%';
                    if (sentimentScoreBar) sentimentScoreBar.style.background = getSentimentColor(data.sentiment.mean_score);
                    if (sentimentScoreValue) sentimentScoreValue.textContent = data.sentiment.mean_score.toFixed(2);
                } else {
                    if (sentimentCard) sentimentCard.style.display = 'none';
                }
                if (data.all_probabilities && Object.keys(data.all_probabilities).length > 0) {
                    const allowedGroups = [
                        'High-Volume Consumer Products',
                        'Credit & Lending Portfolio',
                        'Regulatory & Collections',
                        'Specialized Services',
                        'Banking Services'
                    ];
                    const sortedProbs = Object.entries(data.all_probabilities)
                        .filter(([group, _]) => allowedGroups.includes(group))
                        .sort(([,a],[,b]) => b-a);
                    if (allProbabilitiesListElem) allProbabilitiesListElem.innerHTML = '';
                    sortedProbs.forEach(([product, prob]) => {
                        // Dashboard style row
                        const row = document.createElement('div');
                        row.className = 'probability-row';
                        const label = document.createElement('div');
                        label.className = 'prob-label';
                        label.textContent = product;
                        const barWrap = document.createElement('div');
                        barWrap.className = 'prob-bar-wrap';
                        const bar = document.createElement('div');
                        bar.className = 'prob-bar';
                        const barInner = document.createElement('div');
                        barInner.className = 'prob-bar-inner';
                        bar.appendChild(barInner);
                        barWrap.appendChild(bar);
                        const value = document.createElement('div');
                        value.className = 'prob-value';
                        const percent = (prob * 100).toFixed(2);
                        value.textContent = percent + '%';
                        row.appendChild(label);
                        row.appendChild(barWrap);
                        row.appendChild(value);
                        if (allProbabilitiesListElem) allProbabilitiesListElem.appendChild(row);
                        // Animate bar
                        setTimeout(() => {
                            barInner.style.width = percent + '%';
                        }, 100);
                    });
                    if (allProbabilitiesContainer) allProbabilitiesContainer.style.display = 'block';
                }
            } else {
                 displayError("Received an unexpected response from the server.");
            }
        } catch (error) {
            if (loadingIndicator) loadingIndicator.style.display = 'none';
            console.error('Prediction error:', error);
            displayError(error.message || 'An error occurred while fetching the prediction.');
        }
    });

    function animateConfidenceBar(confidence) {
        if (!confidenceBarInner) return;
        const percent = Math.max(0, Math.min(100, confidence * 100));
        gsap.to(confidenceBarInner, { width: percent + '%', duration: 1.2, ease: 'power3.out' });
    }

    async function typeText(element, text, speed = 20) {
        if (!element) return;
        element.textContent = '';
        for (let i = 0; i < text.length; i++) {
            element.textContent += text[i];
            await new Promise(res => setTimeout(res, speed));
        }
    }

    function displayError(message) {
        if (errorMessageElem) errorMessageElem.textContent = message;
        if (errorSection) errorSection.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';
        if (sentimentCard) sentimentCard.style.display = 'none';
    }

    function getSentimentColor(score) {
        if (score > 0.3) return 'linear-gradient(90deg, #43e97b 0%, #38f9d7 100%)';
        if (score < -0.3) return 'linear-gradient(90deg, #fa709a 0%, #fee140 100%)';
        return 'linear-gradient(90deg, #f7971e 0%, #ffd200 100%)';
    }
    function capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    }
});