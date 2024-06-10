import express from 'express';
import 'dotenv/config';
import http from 'http';
// import cors from 'cors';
import chalk from 'chalk';
import fs from 'fs';
import connectDB from './mongodb/connect.js';
import errorHandler from './middlewares/errorHandler.js';

import candidateRoutes from './routes/candidateRoutes.js';
import questionRoutes from './routes/questionRoutes.js';
import assessmentRoutes from './routes/assessmentRoutes.js';
import responseRoutes from './routes/responseRoutes.js';


const app = express();

app.use(express.json());
app.use(express.urlencoded({extended:true}));


// Ensure uploads directory exists for multer storage
if (!fs.existsSync('./uploads')) {
  fs.mkdirSync('./uploads');
}

connectDB();

app.get('/',(req,res)=>{
    res.json({message:'API is running'})
});

app.use('/api/candidates',candidateRoutes);
app.use('/api/questions',questionRoutes);
app.use('/api/assessments',assessmentRoutes);
app.use('/api/responses',responseRoutes);

app.use(errorHandler);

const server = http.createServer(app);

const port = process.env.PORT || 5000
server.listen(port,()=>{
    console.log(`Server is running on port: ${chalk.cyan(port)}`);
})