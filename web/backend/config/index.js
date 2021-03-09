module.exports = {
  /**
   * Port that server listens at
   */
  port: 3000,

  /**
   *  timeout in millisecond
   */
  timeout: 300000,

  /**
   *  Logging information
   */
  logs: {
    level: "debug",
    format: {
      time: 'YYYY-MM-DD HH:mm:ss'
    }
  },

  /**
   * API routes
   */
  routes: {
    phonetisaurus: 'http://localhost:5001/predict',
    transformer: 'http://localhost:5002/predict'
  }
}
