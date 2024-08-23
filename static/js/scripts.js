document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const textarea = document.querySelector("textarea");
    const resultAlert = document.querySelector(".alert");
    const historyList = document.querySelector(".list-group");

    // Hide result alert initially
    if (resultAlert) {
        resultAlert.style.display = "none";
    }

    // Handle form submission
    form.addEventListener("submit", function(event) {
        event.preventDefault();

        // Show loading spinner
        showLoadingSpinner();

        const formData = new FormData(form);
        const text = formData.get("text");

        // Send the text data to the Flask server using Fetch API
        fetch("/", {
            method: "POST",
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            // Hide loading spinner
            hideLoadingSpinner();

            // Update the DOM with the response (replace the entire body)
            document.body.innerHTML = data;

            // Re-bind event listeners after DOM update
            form = document.querySelector("form");
            resultAlert = document.querySelector(".alert");
            historyList = document.querySelector(".list-group");

            // Show the result alert with animation
            if (resultAlert) {
                resultAlert.style.display = "block";
                resultAlert.classList.add("fade-in");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            hideLoadingSpinner();
            alert("An error occurred. Please try again.");
        });
    });

    function showLoadingSpinner() {
        const spinner = document.createElement("div");
        spinner.className = "spinner-border text-primary";
        spinner.setAttribute("role", "status");
        spinner.innerHTML = `<span class="visually-hidden">Loading...</span>`;
        form.appendChild(spinner);
    }

    function hideLoadingSpinner() {
        const spinner = form.querySelector(".spinner-border");
        if (spinner) {
            spinner.remove();
        }
    }
});
