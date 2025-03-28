document.getElementById('recommendationForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const movieTitle = document.getElementById('movie_title').value;

    fetch('/rec', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `movie_title=${encodeURIComponent(movieTitle)}`,
    })
    .then(response => response.json())
    .then(data => {
        const tableBody = document.querySelector('#recommendationsTable tbody');
        tableBody.innerHTML = ''; // Очищаем таблицу

        if (data.error) {
            alert(data.error);
        } else {
            data.forEach(movie => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${movie.title}</td>
                    <td>${movie.averageRating.toFixed(2)}</td>
                    <td>${movie.year}</td>
                `;
                tableBody.appendChild(row);
            });
        }
    })
    .catch(error => console.error('Ошибка:', error));
});