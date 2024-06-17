import express from 'express';


import { createQuestions } from '../controllers/questionFunctions.js';
const router = express.Router();
router.post('/', createQuestions);

export default router;