const form = document.querySelector('form');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const texto = document.getElementById('input').value;
    const language = document.getElementById('language').value;

    fetch('http://localhost:5000/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: language, texto: texto })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('resposta').innerText = data.resposta;
        if (data.resposta === 'This news is predicted to be FAKE.') {
            document.getElementById('resposta').style.color = 'red'
        } else {
            document.getElementById('resposta').style.color = 'green'
        }
    })
    .catch(err => {
        document.getElementById('resposta').innerText = 'Erro ao conectar com o servidor Python.';
        console.error(err);
    });

});