module.exports = function override(config) {
    config.module.rules.push({
      test: /\.(js|jsx)$/,
      exclude: /node_modules\/(?!@mui)/, // Transpile MUI modules only
      use: {
        loader: 'babel-loader',
        options: {
          presets: ['@babel/preset-env', '@babel/preset-react'],
        },
      },
    });
    return config;
  };