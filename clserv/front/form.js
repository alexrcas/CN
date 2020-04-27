var express = require('express');
var router = express.Router();
var path = require('path');
var fs = require('fs');

const readMetaInfo = (filename) => {return new Promise((res, rej) => {
    const directoryPath = path.join(__dirname, '../cloud');
    fs.readFile(`${directoryPath}/${filename}.json`, (err, data) => {
      res(data);
    })
  })
}


/* GET users listing. */
router.get('/', function(req, res, next) {

    readMetaInfo(req.query.program).then(data => {
        data = JSON.parse(data);
        res.render('form', {
            programName: data.name + '.exe',
            title: data.name,
            desc: data.desc,
            inputs: data.inputs,
            args: data.args
        });
    })
});

module.exports = router;
