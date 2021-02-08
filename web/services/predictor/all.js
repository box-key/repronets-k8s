const axios = require('axios');

module.exports = (input, language, beam) => {
  let wow = {
    input: input,
    language: language,
    beam: beam,
    name: 'Your greed'
  }
  return wow
};
