const axios = require('axios');
const routes = require('../../config').routes;
const logger = require('../../loaders/logger')(module);

module.exports = async function(batch, language, beam) {
  data = {
    batch: batch,
    language: language,
    beam: beam
  };

  const psPrediction = {
    name: 'phonetisaurus',
    request: axios.post(routes.phonetisaurus, data)
  };
  const tsPrediction = {
    name: 'transformer',
    request: axios.post(routes.transformer, data)
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
        logger.debug(`all (${modelNames[i]}) = ${JSON.stringify(responses[i].status)}`)
      };
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
