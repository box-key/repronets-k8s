const express = require('express');
const logger = require('./loaders/logger')(module);
const config = require('./config');
const predict_route = require('./api/routes/predict');
const bodyParser = require('body-parser')

async function startServer() {
  const app = express();
  app.use(bodyParser.urlencoded({ extended: false }))
  app.use(bodyParser.json())

  app.get('/', (req, res) => res.send('Hello World!'))

  app.use('/predict', predict_route())

  let server = app.listen(config.port, () => {
    console.log(`
      #############################################
         Server listening on port: ${config.port}
      #############################################
    `);
  }).on('error', err => {
    logger.error(err);
    process.exit(1);
  });

  server.setTimeout(config.timeout);

}

startServer();
