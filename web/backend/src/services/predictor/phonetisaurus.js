const axios = require('axios');
const routes = require('../../config')[process.env.NODE_ENV || 'development'].routes;
const logger = require('../../loaders/logger')(module);

module.exports = async (input, language, beam) => {

  let params = {
    input: input,
    beam: beam
  };

  let hostHeader = {
    Host: `repronet-phs-${language}.repronets.10.103.195.76.sslip.io`
  };

  return axios.get(routes.phonetisaurus, { params: params, headers: hostHeader })
    .then((resp) => {
      logger.debug(`phs outputs = ${JSON.stringify(resp.status)}`);
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
