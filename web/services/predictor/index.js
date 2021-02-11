const axios = require('axios');
const logger = require('../../loaders/logger')(module);
const phonetisaurus = require('phonetisaurus');
const transformer = require('transformer');
const all = require('all');

module.exports = async (input, language, beam, model) => {
  if (model == 'phonetisaurus') {
    let output = await phonetisaurus(input, language, beam);
  } else if (model == 'transformer') {
    let output = await transformer(input, language, beam);
  } else if (model == 'all') {
    let output = await all(input, language, beam);
  } else {
    logger.error(`Received undefined model = ${model}`)
    output = {
      status: 400,
      message: "undefined model"
    };
  }
  return output;
};
