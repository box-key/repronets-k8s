const axios = require('axios');
const logger = require('../../loaders/logger')(module);

module.exports = async function(input, language, beam) {
  params = {
    input: input,
    language: language,
    beam: beam
  };
  let res = await axios.get('http://localhost:5002/predict', { params: params })
        .catch((err) => { logger.error(err) });
  logger.info(JSON.stringify(res.data))
  return res.data;
};
