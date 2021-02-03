export default {

  port: parseInt(process.env.PORT, 3000)

  logs: {
    level: "debug",
    format: {
      time: 'YYYY-MM-DD HH:mm:ss'
    }
  }
}
