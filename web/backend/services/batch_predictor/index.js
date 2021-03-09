const axios = require('axios');
const logger = require('../../loaders/logger')(module);
const phonetisaurus = require('./phonetisaurus');
const transformer = require('./transformer');
const all = require('./all');

module.exports = async (batch, language, beam, model) => {
  let output = {};

  if (model == 'phonetisaurus') {
    output = await phonetisaurus(batch, language, beam);
  } else if (model == 'transformer') {
    output = await transformer(batch, language, beam);
  } else if (model == 'all') {
    output = await all(batch, language, beam);
  } else {
    logger.error(`Received undefined model = ${model}`)
    output = {
      status: 400,
      message: "undefined model"
    };
  }

  return output;
};
