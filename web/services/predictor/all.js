const axios = require('axios');
const logger = require('../../loaders/logger')(module);

module.exports = async function(input, language, beam) {
  params = {
    input: input,
    language: language,
    beam: beam
  };
  const psPrediction = axios.get('http://localhost:5001/predict', { params: params });
  const tsPrediction = axios.get('http://localhost:5002/predict', { params: params });
  const requests = [
    psPrediction,
    tsPrediction
  ];
  return axios.all(requests)
    .catch(errors => { logger.error(errors) });
};
