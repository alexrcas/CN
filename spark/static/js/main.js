let socket = io.connect('http://localhost:5001', { 'forceNew': true });



socket.on('data', data => {

    data = JSON.parse(data)

    X = []
    Y = []
    data.forEach(item => {
        X.push(item.word)
        Y.push(item.count)
    })

    let plotData = [{
        x: X,
        y: Y,
        type: 'bar'
    }]

    Plotly.newPlot('tester', plotData);
})
