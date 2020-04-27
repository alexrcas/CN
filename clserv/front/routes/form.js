var express = require('express');
var router = express.Router();
const axios = require('axios');

/* GET users listing. */
router.get('/', function(req, res, next) {

  axios.get(`http://localhost:3000/api/form?name=${req.query.service}`)
    .then( response => {

      res.render('form', {
        title: response.data.name,
        desc: response.data.desc,
        inputs: response.data.inputs,
        args: response.data.args
      });


    })
    .catch( err => { res.send(err) } )


});

module.exports = router;
