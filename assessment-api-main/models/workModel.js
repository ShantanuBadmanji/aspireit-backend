import mongoose from "mongoose";


const workSchema = new mongoose.Schema({
    comany:{
        type: String,
        required: true
    },
    type: {
        type: String,
        enum: ['Full Time', 'Part Time', 'Internship'],
        default: 'Full Time'
    },
    industry:{
        type: String,
        enum: ['Software', 'Finance', 'Education', 'Healthcare'],
        default: 'Software'
    },
    duration:{
        type: Number,
        required: true
    },
    candidate_id:{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Candidate',
        required: true
    }
}, {timestamps: true});

const workModel = mongoose.model('Work', workSchema);

export default workModel;