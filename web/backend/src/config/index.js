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
   * Configs depending on NODE_ENV
   */
  development: {
    /**
     * API routes
     */
    routes: {
      phonetisaurus: 'http://192.168.49.2:31388/predict',
      transformer: 'http://192.168.49.2:31388/predict'
    }
  },
  production: {
    /**
     * API routes
     */
    routes: {
      phonetisaurus: 'http://kourier.kourier-system.svc.cluster.local/predict',
      transformer: 'http://kourier.kourier-system.svc.cluster.local/predict'
    }
  }
  
}
