const axios = require('axios');
const logger = require('../../loaders/logger')(module);

module.exports = async function(input, language, beam) {
  params = {
    input: input,
    language: language,
    beam: beam
  };
  return axios.get('http://localhost:5002/predict', { params: params })
    .then((resp) => {
      logger.debug(`ts outputs = ${JSON.stringify(resp.status)}`);
      return resp.data;
    })
    .catch((err) => { 
      logger.error(err) 
    });
};
