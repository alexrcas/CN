var express = require('express');
var router = express.Router();
const axios = require('axios');

/* GET home page. */
router.get('/', function(req, res, next) {
    res.send('api')
});


router.get('/services', function(req, res, next) {
    axios.get('http://localhost:3000/api/services')
        .then(response => {
            res.send(response.data);
        })
        .catch(err => {
            res.send(err);
        })
});


router.post('/exec', function(req, res, next) {

    let program = req.headers.referer.split('?')[1].split('=')[1];
    axios.post('http://localhost:3000/api/exec',
        {
            data: req.body
        }
    )
    .then( response => {res.send(response.data)})
    .catch(err => {console.log(err)})
});

module.exports = router;
