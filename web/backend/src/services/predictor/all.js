const axios = require('axios');
const routes = require('../../config')[process.env.NODE_ENV || 'development'].routes;
const logger = require('../../loaders/logger')(module);

module.exports = async function(input, language, beam) {

  let params = {
    input: input,
    beam: beam
  };

  let hostHeaderPhs = {
    Host: `repronet-phs-${language}.repronets.10.103.195.76.sslip.io`
  };

  let hostHeaderTrf = {
    Host: `repronet-trf-${language}.repronets.10.103.195.76.sslip.io`
  };

  const psPrediction = {
    name: 'phonetisaurus',
    request: axios.get(routes.phonetisaurus, { params: params, headers: hostHeaderPhs })
  };
  const tsPrediction = {
    name: 'transformer',
    request: axios.get(routes.transformer, { params: params, headers: hostHeaderTrf })
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
