const express = require('express');
const logger = require('./loaders/logger')(module);
const config = require('./config');
const predict_route = require('./api/routes/predict');
const bodyParser = require('body-parser')
const cors = require("cors")

async function startServer() {
  const app = express();
 
  /* max payload limit */
  app.use(bodyParser.urlencoded({ extended: false, limit: '20mb' }));
  app.use(bodyParser.json({ limit: '20mb' }));

  /* cors */
  app.use(
    cors({
      origin: "*",
      methods: ["GET", "POST"]
    })
  )

  /* routes */
  app.get('/', (req, res) => res.send('Hello World!'));
  app.use('/predict', predict_route());

  /* start listening */
  let server = app.listen(config.port, () => {
    console.log(`
      #############################################
         Server listening on port: ${config.port}
         Mode: ${process.env.NODE_ENV}
      #############################################
    `);
  }).on('error', err => {
    logger.error(err);
    process.exit(1);
  });

  /* time out */
  server.setTimeout(config.timeout);

}

startServer();
