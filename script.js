document.addEventListener("DOMContentLoaded", () => {
  const inputArea = document.getElementById("article-input");
  const predictButton = document.getElementById("predict-button");
  const resultsSection = document.getElementById("results-section");
  const predictedLabel = document.getElementById("predicted-label");
  const loadingIndicator = document.getElementById("loading-indicator");

  function mockClassification(text) {
    const lowerText = text.toLowerCase();

    const fakeKeywords = [
      "amazing fact",
      "shocking truth",
      "totally fake",
      "exclusive video",

      "liberal media",
      "socialist",
    ];
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

    predictButton.disabled = true;
    resultsSection.classList.remove("hidden");
    predictedLabel.innerText = "";
    predictedLabel.className = "";
    loadingIndicator.classList.remove("hidden");

    setTimeout(() => {
      const result = mockClassification(articleText);
      loadingIndicator.classList.add("hidden");
      predictedLabel.innerText = result;
      predictedLabel.classList.add(result.toLowerCase());
      predictButton.disabled = false;
    }, 1500);
  });
});

