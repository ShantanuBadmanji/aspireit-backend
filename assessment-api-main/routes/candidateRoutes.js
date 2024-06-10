import express from 'express';
import { createCandidate, getCandidates, updateCandidate } from '../controllers/condidateFunctions.js';

const routes = express.Router();

routes.route('/').get(getCandidates).post(createCandidate);
routes.route('/:id').put(updateCandidate);

export default routes;