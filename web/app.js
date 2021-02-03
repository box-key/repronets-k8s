const express = require('express');
const logger = require('./loaders/logger');
const config = require('./config');

async function startServer() {
  const app = express();

  app.get('/', (req, res) => res.send('Hello World!'))
  app.listen(config.port, () => {
    logger.info(`
      #############################################
         Server listening on port: ${config.port} 
      #############################################
    `);
  }).on('error', err => {
    logger.error(err);
    process.exit(1);
  });

}

startServer();
