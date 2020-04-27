var express = require('express');
var router = express.Router();
const axios = require('axios');

/* GET home page. */
router.get('/', function(req, res, next) {

  axios.get('http://localhost:3000/api/services')
        .then(response => {
            res.render('index', {
              services: response.data
            });
        })
        .catch(err => {
            res.send(err);
        });
});

module.exports = router;
