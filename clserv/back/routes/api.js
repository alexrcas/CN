var express = require('express');
var router = express.Router();
var path = require('path');
var fs = require('fs');
var axios = require('axios');
const { exec } = require('child_process');
var multer = require('multer')({
    dest: 'public/uploads'
});


const getServices = () => {return new Promise((res, rej) => {
    let programs = [];
    const directoryPath = path.join(__dirname, '../cloud');
    fs.readdir(directoryPath, (err, files) => {
        files = files.map(file => { return file.split('.')[0] });
        files = files.filter((item, index) => {return files.indexOf(item) === index})
        res(files);
    });
  });
}

let paramsToString = paramsObject => {
    let paramString = '';
    for (param in paramsObject) {
       if(paramsObject[param] != '') {
            if (paramsObject[param] == 'checked')
                paramString += `${param} `
            else
                paramString += `${param} ${paramsObject[param]} `
        }
    }

    return paramString;
}

/* GET home page. */
router.get('/', function(req, res, next) {
    res.send('api');
});


router.get('/services', function(req, res, next) {
    getServices().then(services => {
        res.send(services);
    })
});


router.get('/form', function(req, res, next) {
    res.sendFile(path.join(__dirname, `../cloud/${req.query.name}.json`));
});


router.post('/exec', function(req, res, next) {
    console.log('form!')
    if (req.files) {
        console.log('fichero!')
    }
    res.json({"status": "ok"})
    /*
    console.log('post!')
    let data = (req.body.data)
    console.log(data);
    let program = data.service
    delete data.service
    let params = paramsToString(data);
    let command = `.\\cloud\\${program}.exe ${params}`

    exec(command, (err, stdout, stderr) => {
        if(err) {
            console.log('no se pudo lanzar!')
        }
        res.send(stdout);
    })

    res.send('ok');*/

});


module.exports = router;
