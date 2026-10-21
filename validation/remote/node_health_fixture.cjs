// Disposable validation fixture only. Never COPY into an image or deploy.
// Simulates readiness; does not prove MongoDB/Atlas connectivity.
const mongoose = require('/app/node_modules/mongoose');
mongoose.set('autoCreate', false);
mongoose.set('autoIndex', false);
mongoose.connect = async () => { mongoose.connection._readyState = 1; return mongoose; };
mongoose.disconnect = async () => { mongoose.connection._readyState = 0; };
