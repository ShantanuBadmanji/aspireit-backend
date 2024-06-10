import mongoose from "mongoose";

const analysisSchema = new mongoose.Schema({
    response_id:{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Response',
        required: true
    },
    candidate_id:{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Candidate',
        required: true
    },
    score:{
        type: Number,
        required: true
    },
    total:{
        type: Number,
        required: true
    },
    duration:{
        type: Number,
        required: true
    }
},{timestamps: true});

const analysisModel = mongoose.model('Analysis', analysisSchema);

export default analysisModel;