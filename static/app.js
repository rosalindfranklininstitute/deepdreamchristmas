
class HostApp extends React.Component {

    state = {
        images: []
    };

    componentDidMount() {
        fetch("/images")
            .then(res => res.json()).then(
                (result) => {
                    this.setState({
                        images: result.images
                    });
                },
                (error) => {
                    console.log(error);
                });

    }

    render() {

        return (
            <div id="app-root" className="fill">
                <div className="d-flex h-100 text-center text-white">
                    <div id="root-ui" className="cover-container d-flex w-100 h-100 p-3 mx-auto flex-column">

                        <header>
                            <div>
                                <h3 className="float-md-start mb-0">RFI Christmas Dream</h3>
                                <nav className="nav nav-masthead justify-content-center float-md-end">
                                    <button type="button" className="btn btn-light" data-bs-toggle="modal" data-bs-target="#uploadModal">Upload</button>
                                    <a className="nav-link" aria-current="page" href="#">Upload</a>
                                    <a className="nav-link active" href="/">Gallery</a>
                                    <a className="nav-link" href="/about">About</a>
                                </nav>
                            </div>
                        </header>
                        <div className="container">

                            <div className="modal fade" id="uploadModal" tabIndex="-1" role="dialog"
                             aria-labelledby="uploadModalLabel" aria-hidden="true">
                                <div className="modal-dialog" role="document">
                                    <div className="modal-content">
                                        <div className="modal-header">
                                            <h5 className="modal-title" id="uploadModalLabel">Upload Image</h5>
                                            <button type="button" className="close" data-dismiss="modal" aria-label="Close">
                                                <span aria-hidden="true">&times;</span>
                                            </button>
                                        </div>
                                        <div className="modal-body">
                                            ...
                                        </div>
                                        <div className="modal-footer">
                                            <button type="button" className="btn btn-secondary" data-dismiss="modal">Close</button>
                                            <button type="button" className="btn btn-primary">Upload</button>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="row">
                                {this.state.images.map((image) => <div className="col-md-4 col-xs-12"><img className="img-thumbnail mt-1 mb-1" alt="" src={image}/></div>)}
                            </div>
                        </div>
                        <footer className="text-white-50">
                            <p></p>
                        </footer>
                    </div>
                </div>
            </div>
        );
    }
}

ReactDOM.render(<HostApp />, document.getElementById('root'));
