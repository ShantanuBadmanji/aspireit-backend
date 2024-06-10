import asyncHandler from "express-async-handler";
import Candidate from "../models/candidateModel.js";


const createCandidate = asyncHandler(async (req, res) => {
    const {name,contact_no,email,domain,resume,links} = req.body;

    if(!name || !contact_no || !email){
        res.status(400);
        throw new Error('Insufficient details provided');
    }

    const emailExists = await Candidate.findOne({email});
    const contactExists = await Candidate.findOne({contact_no});

    if(emailExists || contactExists){
        res.status(400);
        throw new Error('email or contact number already exists');
    }

    const newCandidate = Candidate({name,contact_no,email,domain,resume,links});

    const savedCandidate = await newCandidate.save();

    if(savedCandidate){
        res.status(200).json({
            _id: savedCandidate._id,
            name: savedCandidate.name,
            email: savedCandidate.email,
            domain: savedCandidate.domain,
            // resume: savedCandidate.resume,
            links: savedCandidate.links
        });
    }else{
        res.status(400);
        throw new Error("Unable to create new user");
    }
});

const getCandidates = asyncHandler(async (req, res) => {
    const candidates = await Candidate.find({});
    if(candidates){
        res.status(200).json(candidates);
    }else{
        res.status(400);
        throw new Error('No candidates found');
    }
});

const updateCandidate = asyncHandler(async (req, res) => {
    const newInfo = req.body;
    const candidateId = req.params.id;

    const updateCandidate = await Candidate.findByIdAndUpdate(candidateId, newInfo, {new:true});
    if(updateCandidate){
        res.status(200).json(updateCandidate);
    }else{
        res.status(400);
        throw new Error('Unable to update candidate');
    }
});

export {createCandidate, getCandidates, updateCandidate};