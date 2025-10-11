document.addEventListener("DOMContentLoaded", () => {
  const inputArea = document.getElementById("article-input");
  const predictButton = document.getElementById("predict-button");
  const resultsSection = document.getElementById("results-section");
  const predictedLabel = document.getElementById("predicted-label");
  const loadingIndicator = document.getElementById("loading-indicator");

  // Simple function to decide a *mock* result based on keywords
  function mockClassification(text) {
    const lowerText = text.toLowerCase();

    // Keywords that bias the result towards "FAKE"
    const fakeKeywords = [
      "amazing fact",
      "shocking truth",
      "totally fake",
      "exclusive video",
      "liberal media",
      "socialist",
    ];

    // Keywords that bias the result towards "REAL"
    const realKeywords = [
      "according to",
      "spokesperson",
      "president",
      "reported",
      "stock market",
      "government official",
    ];

    let score = 0;

    fakeKeywords.forEach((word) => {
      if (lowerText.includes(word)) {
        score -= 1;
      }
    });

    realKeywords.forEach((word) => {
      if (lowerText.includes(word)) {
        score += 1;
      }
    });

    // Use a slight random element to make it less predictable
    score += Math.random() - 0.5;

    return score > 0 ? "REAL" : "FAKE";
  }

  predictButton.addEventListener("click", () => {
    const articleText = inputArea.value.trim();

    if (articleText.length < 50) {
      alert(
        "Please enter a longer article (at least 50 characters) for classification."
      );
      return;
    }

    // 1. Start Simulation: Disable button, show loading
    predictButton.disabled = true;
    resultsSection.classList.remove("hidden");
    predictedLabel.innerText = "";
    predictedLabel.className = "";
    loadingIndicator.classList.remove("hidden");

    // 2. Simulate Backend Processing (calling the C program)
    // In a real application, this would be an AJAX/Fetch call to a server endpoint
    // that executes your C program (e.g., via CGI, FastCGI, or a WASM bridge).
    setTimeout(() => {
      const result = mockClassification(articleText);

      // 3. Display Results
      loadingIndicator.classList.add("hidden");
      predictedLabel.innerText = result;
      predictedLabel.classList.add(result.toLowerCase());
      predictButton.disabled = false;
    }, 1500); // Simulated delay of 1.5 seconds
  });
});
