// Get the form element
const form = document.querySelector('form');

// Add an event listener to the form submission
form.addEventListener('submit', (event) => {
  // Prevent the default form submission
  event.preventDefault();

  // Get the file input element
  const fileInput = document.querySelector('input[type="file"]');

  // Check if a file was selected
  if (fileInput.files.length > 0) {
    // Get the file object
    const file = fileInput.files[0];

    // Check if the file is a PDF
    if (file.type === 'application/pdf') {
      // Create a new FormData object
      const formData = new FormData();

      // Add the file to the form data
      formData.append('resume', file);

      // Send a POST request to the server to check the resume
      fetch('/check_resume', {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((result) => {
          // Display the result to the user
          alert(result.message);
        })
        .catch((error) => {
          console.error(error);
          alert('An error occurred while checking the resume.');
        });
    } else {
      alert('Please upload a PDF file.');
    }
  } else {
    alert('Please select a file to upload.');
  }
});


