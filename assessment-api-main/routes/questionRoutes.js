import express from 'express';
import { getQuestions, createQuestion, deleteQuestion } from '../controllers/questionFunctions.js';

const routes = express.Router();

routes.route('/').get(getQuestions).post(createQuestion);
routes.route('/:id').delete(deleteQuestion);

export default routes;