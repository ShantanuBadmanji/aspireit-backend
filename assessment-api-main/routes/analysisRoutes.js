import express from 'express';
import {receiveData} from '../controllers/analysisFunctions.js';

const router = express.Router();

router.post('/receive-data', receiveData)

export default router;


