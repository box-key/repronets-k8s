const axios = require('axios');
const routes = require('../../config')[process.env.NODE_ENV || 'development'].routes;
const logger = require('../../loaders/logger')(module);

module.exports = async function(input, language, beam) {

  let params = {
    input: input,
    beam: beam
  };

  let hostHeader = {
    Host: `repronet-trf-${language}.repronets.example.com`
  };

  return axios.get(routes.transformer, { params: params, headers: hostHeader })
    .then((resp) => {
      logger.debug(`trf outputs = ${JSON.stringify(resp.status)}`);
      return resp.data;
    })
    .catch((err) => {
      logger.error(err);
      return {
      	resp: 500,
      	message: err
      };
    });
};
