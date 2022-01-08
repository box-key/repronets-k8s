const config = require('../config');
const winston = require('winston');
const { combine, timestamp, label, printf } = winston.format;
const path = require('path');

const myFormat = printf(({ level, message, label, timestamp }) => {
  return `[${timestamp} ${label} ${level.toUpperCase()}] ${message}`;
});

const getLabel = function(callingModule) {
  const parts = callingModule.filename.split(path.sep);
  return path.join(parts[parts.length - 2], parts.pop());
};

module.exports = function(callingModule) {
  return winston.createLogger({
    level: config.logs.level,
    levels: winston.config.npm.levels,
    format: combine(
      label({ label: getLabel(callingModule)}),
      timestamp({
        format: config.logs.format.time
      }),
      winston.format.splat(),
      myFormat,
    ),
    transports: [new winston.transports.Console()]
  });
};
