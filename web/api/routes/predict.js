const logger = require('../../loaders/logger')(module);
const phonetisaurus = require('../../services/predictor/phonetisaurus');
const transformer = require('../../services/predictor/transformer');
const all = require('../../services/predictor/all');
const express = require('express');
const router = express.Router();

router.get('/', async function(req, res) {
  let input = req.query.input;
  let language = req.query.language;
  let model = req.query.model;
  let beam = req.query.beam;
  logger.info(`Request body = ${JSON.stringify(req.query)}`);
  if (input == undefined) {
    res.send('Bad boy!')
  } else if (model == 'phonetisaurus') {
    let output = await phonetisaurus(input, language, beam);
    logger.info(`output = ${JSON.stringify(output)}`);
    res.json(output).status(200);
  } else if (model == 'transformer') {
    let output = await transformer(input, language, beam);
    logger.info(`output = ${JSON.stringify(output)}`);
    res.json(output).status(200);
  } else if (model == 'all') {
    let output = all(input, language, beam);
    res.json(output).status(200);
  } else {
    res.send('No Model!')
  }
});

module.exports = router;
