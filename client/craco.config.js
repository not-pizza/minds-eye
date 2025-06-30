module.exports = {
  // We're keeping the basic CRACO configuration in case we need
  // to add custom webpack configurations in the future
  webpack: {
    configure: (webpackConfig) => {
      return webpackConfig;
    },
  },
};