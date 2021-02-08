const axios = require('axios');

module.exports = (input, language, beam) => {
  params = {
    input: input,
    language: language,
    beam: beam
  };
  let res = await axios.get('http://localhost:5001', { params })
  return res.data;
};
