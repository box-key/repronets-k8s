const axios = require('axios');

module.exports = (input, language, beam) => {
  let wow = {
    input: input,
    language: language,
    beam: beam,
    name: 'transformer'
  }
  return wow
};
