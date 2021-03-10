const logger = require('../../loaders/logger')(module);
const predictor = require('../../services/predictor');
const batch_predictor = require('../../services/batch_predictor');
const express = require('express');
const router = express.Router();
const {check, query, body, validationResult} = require('express-validator');

module.exports = () => {
  /*
   * Define sets to check input values
   */
  let supportedLangs = new Set([
    'ara',
    'chi',
    'heb',
    'jpn',
    'kor',
    'rus'
  ])
  let supportedModels = new Set([
    'phonetisaurus',
    'transformer',
    'all'
  ])

  /*
   * Define GET method. It's intended for a real time prediction.
   */
  router.get('/',[
    query('input', `Input must be a string less than 37 characters`)
      .notEmpty()
      .isString()
      .isLength({ min: 1, max: 36 }),
    query('language', `Language must be a string. The list of supported langauges = [${Array.from(supportedLangs)}]`)
      .notEmpty()
      .isString()
      .isLength({ min: 3, max: 3})
      .custom((language) => supportedLangs.has(language)),
    query('model', `Model must be a string. The list of models = [${Array.from(supportedModels)}]`)
      .notEmpty()
      .isString()
      .custom((model) => supportedModels.has(model)),
    query('beam', `Beam must an integer from 1 to 5`)
      .notEmpty()
      .isInt({ min: 1, max: 5 }),
  ],
  async function(req, res) {
    const errors = validationResult(req);
    if (errors.isEmpty()) {
      let input = req.query.input;
      let language = req.query.language;
      let model = req.query.model;
      let beam = req.query.beam;
      logger.info(`Request query = ${JSON.stringify(req.query)}`);
      let output = await predictor(input, language, beam, model);
      logger.info(`output = ${JSON.stringify(output)}`);
      res.json({
	"data": output,
	"status": 200
      }).status(200);
    } else {
      res.json({
	"message": errors.array()[0],
	"status": 400
      }).status(400);
    }
  });

  /*
   * Define POST method. It's intended for a batch prediction.
   */
  router.post('/',[
    body('language', `Language must be a string. The list of supported langauges = [${Array.from(supportedLangs)}]`)
      .notEmpty()
      .isString()
      .isLength({ min: 3, max: 3})
      .custom((language) => supportedLangs.has(language)),
    body('model', `Model must be a string. The list of models = [${Array.from(supportedModels)}]`)
      .notEmpty()
      .isString()
      .custom((model) => supportedModels.has(model)),
    body('beam', `Beam must an integer from 1 to 5`)
      .notEmpty()
      .isInt({ min: 1, max: 5 }),
    body('batch', `Batch must be an array with less than 100000 elements. Each element must have 'idx' and 'src' fields.`)
      .exists()
      .isArray({ min: 1, max: 100000})
      .custom((batch) => 'idx' in batch[0] && 'src' in batch[0]),
  ],
  async function(req, res) {
    const errors = validationResult(req);
    if (errors.isEmpty()) {
      let batch = req.body.batch;
      let language = req.body.language;
      let model = req.body.model;
      let beam = req.body.beam;
      logger.info(`Request body: language=${language} - model=${model} - beam=${beam} - batch len = ${batch.length}`);
      let output = await batch_predictor(batch, language, beam, model);
      logger.info(`output = ${JSON.stringify(output)}`);
      res.json({
      	"data": output,
      	"status": 200
      }).status(200);
    } else {
      res.json({
      	"message": errors.array()[0],
      	"status": 400
      }).status(400);
    }
  });
  
  /*
   * Return router object
   */
  return router
};
