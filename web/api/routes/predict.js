const logger = require('../../loaders/logger')(module);
const predictor = require('../../services/predictor');
const express = require('express');
const router = express.Router();
const {check, query, validationResult} = require('express-validator');

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
   * Define logic for this route
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
      res.json(output);
    } else {
      res.json(errors.array()[0]).status(400);
    }
  });
  return router
};
