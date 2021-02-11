const axios = require('axios');
const logger = require('../../loaders/logger')(module);

module.exports = async function(input, language, beam) {
  params = {
    input: input,
    language: language,
    beam: beam
  };
  const psPrediction = {
    name: 'phonetisaurus',
    request: axios.get('http://localhost:5001/predict', { params: params })
  };
  const tsPrediction = {
    name: 'transformer',
    request: axios.get('http://localhost:5002/predict', { params: params })
  };
  const modelNames = [
    psPrediction['name'],
    tsPrediction['name']
  ];
  const requests = [
    psPrediction['request'],
    tsPrediction['request']
  ];
  return axios.all(requests)
    .then(axios.spread((...responses) => {
      let model_outputs = {};
      for(i = 0; i < requests.length; i++){
	model_outputs[modelNames[i]] = responses[i].data;
      };
      logger.debug(`all outputs = ${JSON.stringify(model_outputs)}`);
      return model_outputs;
    }))
    .catch((errors) => { 
      logger.error(errors);
      return {
	resp: 500,
	message: errors
      };
    });
};
