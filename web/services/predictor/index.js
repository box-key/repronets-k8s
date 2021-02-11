const axios = require('axios');
const logger = require('../../loaders/logger')(module);
const phonetisaurus = require('./phonetisaurus');
const transformer = require('./transformer');
const all = require('./all');

module.exports = async (input, language, beam, model) => {
  let output = {};
  if (model == 'phonetisaurus') {
    output = await phonetisaurus(input, language, beam);
  } else if (model == 'transformer') {
    output = await transformer(input, language, beam);
  } else if (model == 'all') {
    output = await all(input, language, beam);
  } else {
    logger.error(`Received undefined model = ${model}`)
    output = {
      status: 400,
      message: "undefined model"
    };
  }
  return output;
};
